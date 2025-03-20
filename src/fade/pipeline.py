"""
Document Processing Pipeline

This architecture implements the document processing pipeline using LangGraph for orchestration
and various document processing libraries.
"""

import os
import shutil
from dataclasses import dataclass
from typing import Any

# Core document processing
import fitz  # type: ignore # PyMuPDF

# Layout analysis and OCR
import layoutparser as lp # type: ignore

# LangGraph for orchestration
from langgraph.graph import END, StateGraph # type: ignore
from paddleocr import PaddleOCR # type: ignore


@dataclass
class Entity:
    """Represents a detected entity in a document."""

    entity_id: str
    page_num: int
    bbox: list[float]  # [x0, y0, x1, y1]
    entity_type: str | None = None
    content: Any | None = None
    confidence: float = 0.0
    document_type: str | None = None
    processed: bool = False


@dataclass
class PipelineState:
    """State for the document processing pipeline."""

    document_id: str
    working_dir: str
    images: list[str] = None
    entities: dict[str, Entity] = None
    unclassified_entities: dict[str, Entity] = None
    logs: list[dict] = None
    error: str | None = None


# Initialize document processing tools
ocr = PaddleOCR(use_angle_cls=True, lang="en")
layout_model = lp.AutoLayoutModel("lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
                                 extra_config={"enforce_cpu": True,
                                             "max_size": 1600})


def setup_working_directory(state: PipelineState) -> PipelineState:
    """
    Copy the document folder or file to a working directory.
    """
    try:
        source = state.document_id
        target_dir = f"{source}_working"

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # Handle both file and directory cases
        if os.path.isfile(source):
            # If source is a file, just copy it
            target_file = os.path.join(target_dir, os.path.basename(source))
            shutil.copy2(source, target_file)
        elif os.path.isdir(source):
            # If source is a directory, copy all contents
            for item in os.listdir(source):
                s = os.path.join(source, item)
                t = os.path.join(target_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, t)
                else:
                    shutil.copy2(s, t)
        else:
            state.error = f"Source path {source} does not exist"
            return state

        state.working_dir = target_dir
        state.logs = []
        state.entities = {}
        state.unclassified_entities = {}

        # Log the operation
        state.logs.append(
            {
                "step": "setup_working_directory",
                "status": "success",
                "message": f"Created working directory: {target_dir}",
            }
        )

        return state
    except Exception as e:
        state.error = f"Error in setup_working_directory: {e!s}"
        return state


def extract_document_pages(state: PipelineState) -> PipelineState:
    """
    Extract each page of the document and attachments as images.
    """
    try:
        working_dir = state.working_dir
        images_dir = os.path.join(working_dir, "images")

        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        state.images = []

        # Process PDF files in the working directory
        for filename in os.listdir(working_dir):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(working_dir, filename)

                # Open the PDF with PyMuPDF
                doc = fitz.open(pdf_path)

                # Extract each page as an image
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(alpha=False)
                    image_path = os.path.join(images_dir, f"{filename}_page_{page_num + 1}.png")
                    pix.save(image_path)
                    state.images.append(image_path)

                # Extract attachments if present
                try:
                    # Get number of embedded files
                    embedded_files = doc.embfile_get_names()
                    if embedded_files:
                        for attachment_name in embedded_files:
                            try:
                                # Extract the embedded file
                                attachment_data = doc.embfile_get(attachment_name)
                                attachment_path = os.path.join(working_dir, attachment_name)

                                with open(attachment_path, "wb") as f:
                                    f.write(attachment_data)

                                # If attachment is PDF, process its pages too
                                if attachment_name.lower().endswith(".pdf"):
                                    attachment_doc = fitz.open(attachment_path)
                                    for page_num in range(len(attachment_doc)):
                                        page = attachment_doc.load_page(page_num)
                                        pix = page.get_pixmap(alpha=False)
                                        image_path = os.path.join(images_dir, f"{attachment_name}_page_{page_num + 1}.png")
                                        pix.save(image_path)
                                        state.images.append(image_path)
                                    attachment_doc.close()
                            except Exception as e:
                                print(f"Error processing attachment {attachment_name}: {str(e)}")
                                continue
                except Exception as e:
                    print(f"Error processing attachments in {filename}: {str(e)}")

                doc.close()

        # Log the operation
        state.logs.append(
            {
                "step": "extract_document_pages",
                "status": "success",
                "message": f"Extracted {len(state.images)} page images",
            }
        )

        return state
    except Exception as e:
        state.error = f"Error in extract_document_pages: {e!s}"
        return state


def detect_entities(state: PipelineState) -> PipelineState:
    """
    Detect entities on each page using PaddleOCR with optimized settings.
    """
    try:
        if not state.images:
            state.error = "No images found to process"
            return state

        # Create visualization directory
        vis_dir = os.path.join(state.working_dir, "visualization")
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        # Initialize OCR with heavily optimized settings
        from paddleocr import PaddleOCR # type: ignore
        ocr = PaddleOCR(
            use_angle_cls=False,  # Disable angle detection
            lang='en',
            use_gpu=False,
            show_log=False,
            det_db_thresh=0.3,
            det_db_box_thresh=0.3,
            det_limit_side_len=960,  # Limit image size
            det_limit_type='max',
            rec_batch_num=10,  # Increase batch size
            use_mp=True,  # Enable multiprocessing
            total_process_num=4,  # Number of processes
            cls_batch_num=6,
            rec_algorithm='SVTR_LCNet',  # Faster recognition model
            det_db_score_mode='fast',
            use_dilation=False,
            det_db_unclip_ratio=1.5
        )

        from tqdm import tqdm # type: ignore
        import cv2 # type: ignore
        import numpy as np # type: ignore

        # Process each image with progress bar
        for idx, image_path in enumerate(tqdm(state.images, desc="Processing pages")):
            try:
                # Extract page number from filename
                filename = os.path.basename(image_path)
                page_num = int(filename.split("_page_")[1].split(".")[0])
                
                print(f"\nProcessing page {page_num} ({idx + 1}/{len(state.images)})")

                # Load and preprocess image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue

                # Resize large images
                height, width = image.shape[:2]
                if max(height, width) > 960:
                    scale = 960 / max(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    image = cv2.resize(image, (new_width, new_height))
                    print(f"Resized from {width}x{height} to {new_width}x{new_height}")

                # Process with OCR
                import time
                start_time = time.time()
                result = ocr.ocr(image, cls=False)  # Disable text direction classification
                processing_time = time.time() - start_time
                print(f"OCR processing time: {processing_time:.2f} seconds")

                if not result or not result[0]:
                    print(f"No text detected on page {page_num}")
                    continue

                print(f"Found {len(result[0])} text elements")

                # Process detected elements
                vis_image = image.copy()
                for i, line in enumerate(result[0]):
                    entity_id = f"entity_{filename}_{i}"
                    bbox = line[0]
                    text = line[1][0]
                    confidence = float(line[1][1])

                    # Convert bbox coordinates
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    bbox_rect = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

                    # Create entity
                    entity = Entity(
                        entity_id=entity_id,
                        page_num=page_num,
                        bbox=bbox_rect,
                        confidence=confidence,
                        entity_type="text",
                        content=text,
                        document_type="TXT"
                    )
                    state.entities[entity_id] = entity

                    # Draw visualization
                    points = np.array(bbox, np.int32)
                    cv2.polylines(vis_image, [points], True, (0, 255, 0), 2)

                # Save visualization
                vis_path = os.path.join(vis_dir, f"detected_text_page_{page_num}.png")
                cv2.imwrite(vis_path, vis_image)

            except Exception as page_error:
                print(f"Error processing page {page_num}: {str(page_error)}")
                continue

        state.logs.append({
            "step": "detect_entities",
            "status": "success",
            "message": f"Processed {len(state.images)} pages, detected {len(state.entities)} entities"
        })

        return state
    except Exception as e:
        state.error = f"Error in detect_entities: {e!s}"
        return state


def classify_entities(state: PipelineState) -> PipelineState:
    """
    Classify the type of each entity using layout detection and OCR.
    """
    try:
        if not state.entities:
            state.error = "No entities found to classify"
            return state

        # Maps from layout detection categories to our entity types
        category_mapping = {
            "Text": "text",
            "Title": "text",
            "List": "text",
            "Table": "data",
            "Figure": "image",
            0: "text",  # PubLayNet mapping
            1: "text",  # Title
            2: "text",  # List
            3: "data",  # Table
            4: "image",  # Figure
        }

        # Maps entity types to document types
        document_type_mapping = {"text": "TXT", "image": "PNG", "data": "CSV", "plot": "CSV", "diagram": "PNG"}

        # Process each entity
        for entity_id, entity in state.entities.items():
            # Get the image path corresponding to this entity's page
            image_path = None
            for img in state.images:
                if f"_page_{entity.page_num}." in img:
                    image_path = img
                    break

            if not image_path:
                continue

            # Load image
            image = lp.load_image(image_path)

            # Extract just the entity region from the image
            x0, y0, x1, y1 = entity.bbox
            entity_image = image[int(y0) : int(y1), int(x0) : int(x1)]

            # Try to determine entity type from layout detection
            try:
                # Use PaddleOCR to check if this region contains text
                ocr_result = ocr.ocr(entity_image)
                has_text = len(ocr_result) > 0 and len(ocr_result[0]) > 0

                if has_text:
                    # Extract text content
                    text_content = " ".join([line[1][0] for line in ocr_result[0]])
                    entity.content = text_content
                    entity.entity_type = "text"
                else:
                    # Analyze the image to detect if it's a table, plot, or diagram
                    # For simplicity, we'll use heuristics based on pixel analysis
                    # In a real system, you'd use more sophisticated ML models

                    # Default to image if we can't determine more specifically
                    entity.entity_type = "image"
            except Exception:
                # If classification fails, mark as unclassified
                state.unclassified_entities[entity_id] = entity
                entity.entity_type = None
                continue

            # Set document type based on entity type
            if entity.entity_type in document_type_mapping:
                entity.document_type = document_type_mapping[entity.entity_type]

        # Count classified vs unclassified
        classified_count = len(state.entities) - len(state.unclassified_entities)

        # Log the operation
        state.logs.append(
            {
                "step": "classify_entities",
                "status": "success",
                "message": f"Classified {classified_count} entities, {len(state.unclassified_entities)} remain unclassified",
            }
        )

        return state
    except Exception as e:
        state.error = f"Error in classify_entities: {e!s}"
        return state


def report_unclassified_entities(state: PipelineState) -> PipelineState:
    """
    Reports unclassified entities to be shown to the user.
    """
    try:
        # If there are no unclassified entities, we can skip user input
        if not state.unclassified_entities:
            state.logs.append(
                {
                    "step": "report_unclassified_entities",
                    "status": "success",
                    "message": "No unclassified entities to report",
                }
            )
            return state

        # In a real application, this would prepare data for the UI
        # to display unclassified entities to the user

        state.logs.append(
            {
                "step": "report_unclassified_entities",
                "status": "success",
                "message": f"Reported {len(state.unclassified_entities)} unclassified entities for user input",
            }
        )

        # In LangGraph, we would typically return the state here, and the UI would
        # handle showing the unclassified entities to the user
        return state
    except Exception as e:
        state.error = f"Error in report_unclassified_entities: {e!s}"
        return state


def process_user_input(state: PipelineState, user_classifications: dict[str, str]) -> PipelineState:
    """
    Process user input for unclassified entities.

    Args:
        state: The current pipeline state
        user_classifications: Dictionary mapping entity_ids to their classifications
    """
    try:
        if not user_classifications:
            state.logs.append(
                {"step": "process_user_input", "status": "warning", "message": "No user classifications provided"}
            )
            return state

        # Update entity classifications based on user input
        for entity_id, classification in user_classifications.items():
            if entity_id in state.unclassified_entities:
                entity = state.unclassified_entities[entity_id]
                entity.entity_type = classification

                # Update document type based on entity type
                document_type_mapping = {"text": "TXT", "image": "PNG", "data": "CSV", "plot": "CSV", "diagram": "PNG"}
                if classification in document_type_mapping:
                    entity.document_type = document_type_mapping[classification]

                # Move from unclassified to classified
                state.entities[entity_id] = entity
                del state.unclassified_entities[entity_id]

        state.logs.append(
            {
                "step": "process_user_input",
                "status": "success",
                "message": f"Processed user classifications for {len(user_classifications)} entities",
            }
        )

        return state
    except Exception as e:
        state.error = f"Error in process_user_input: {e!s}"
        return state


def process_entities(state: PipelineState) -> PipelineState:
    """
    Process entities according to their type and create output files.
    """
    try:
        if not state.entities:
            state.error = "No entities to process"
            return state

        output_dir = os.path.join(state.working_dir, "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create JSON structure for all entities
        json_structure = []

        for entity_id, entity in state.entities.items():
            if not entity.entity_type:
                continue

            # Get the image path corresponding to this entity's page
            image_path = None
            for img in state.images:
                if f"_page_{entity.page_num}." in img:
                    image_path = img
                    break

            if not image_path:
                continue

            # Load the page image
            image = lp.load_image(image_path)

            # Extract entity region
            x0, y0, x1, y1 = entity.bbox
            entity_image = image[int(y0) : int(y1), int(x0) : int(x1)]

            # Process based on entity type
            output_path = None

            if entity.entity_type == "text":
                # For text entities, use OCR to extract text
                ocr_result = ocr.ocr(entity_image)
                if ocr_result and ocr_result[0]:
                    text_content = "\n".join([line[1][0] for line in ocr_result[0]])
                    output_path = os.path.join(output_dir, f"{entity_id}.txt")
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(text_content)
                    entity.content = text_content

            elif entity.entity_type == "data" or entity.entity_type == "plot":
                # For table data, try to extract as CSV
                # In a real implementation, we would use Camelot for table extraction
                # This is simplified for demonstration
                output_path = os.path.join(output_dir, f"{entity_id}.csv")

                # Save entity image temporarily for table extraction
                temp_image_path = os.path.join(output_dir, f"{entity_id}_temp.png")
                import cv2 # type: ignore

                cv2.imwrite(temp_image_path, entity_image)

                # In a real implementation:
                # tables = camelot.read_pdf(pdf_path, pages=str(entity.page_num),
                #                          table_areas=[','.join(map(str, entity.bbox))])
                # if tables and len(tables) > 0:
                #     tables[0].to_csv(output_path)

                # Dummy CSV content for demonstration
                with open(output_path, "w") as f:
                    f.write("column1,column2,column3\nvalue1,value2,value3\n")

                # Clean up temp file
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)

            elif entity.entity_type in ["image", "diagram"]:
                # Save as PNG
                output_path = os.path.join(output_dir, f"{entity_id}.png")
                import cv2 # type: ignore

                cv2.imwrite(output_path, entity_image)

            # Add to JSON structure
            json_entity = {
                "entity_id": entity_id,
                "page_num": entity.page_num,
                "bbox": entity.bbox,
                "entity_type": entity.entity_type,
                "document_type": entity.document_type,
                "output_path": output_path.replace(state.working_dir, "").lstrip("/\\") if output_path else None,
            }

            if entity.entity_type == "text" and entity.content:
                json_entity["content"] = entity.content

            json_structure.append(json_entity)

            # Mark as processed
            entity.processed = True

        # Write the JSON structure to a file
        import json

        json_output_path = os.path.join(output_dir, "document_structure.json")
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(json_structure, f, indent=2)

        state.logs.append(
            {
                "step": "process_entities",
                "status": "success",
                "message": f"Processed {len(json_structure)} entities and created output files",
            }
        )

        return state
    except Exception as e:
        state.error = f"Error in process_entities: {e!s}"
        return state


def log_process(state: PipelineState) -> PipelineState:
    """
    Log the processing for ML algorithm improvement.
    """
    try:
        log_dir = os.path.join(state.working_dir, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Write logs to a file
        import json
        from datetime import datetime

        log_filename = f"process_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        log_path = os.path.join(log_dir, log_filename)

        log_data = {
            "document_id": state.document_id,
            "timestamp": datetime.now().isoformat(),
            "entities_count": len(state.entities),
            "unclassified_count": len(state.unclassified_entities),
            "processing_steps": state.logs,
        }

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)

        state.logs.append({"step": "log_process", "status": "success", "message": f"Created process log at {log_path}"})

        return state
    except Exception as e:
        state.error = f"Error in log_process: {e!s}"
        return state


def decide_if_user_input_needed(state: PipelineState) -> str:
    """
    Decide if user input is needed for unclassified entities.
    """
    if state.error:
        return "error"
    elif state.unclassified_entities:
        return "need_user_input"
    else:
        return "continue_processing"


def handle_error(state: PipelineState) -> PipelineState:
    """
    Handle errors in the pipeline.
    """
    print(f"Error in pipeline: {state.error}")
    return state


# Define the LangGraph workflow
def create_document_processing_graph():
    """
    Create a LangGraph workflow for document processing.
    """
    # Create StateGraph with PipelineState
    workflow = StateGraph(PipelineState)

    # Define the nodes
    workflow.add_node("setup_working_directory", setup_working_directory)
    workflow.add_node("extract_document_pages", extract_document_pages)
    workflow.add_node("detect_entities", detect_entities)
    workflow.add_node("classify_entities", classify_entities)
    workflow.add_node("report_unclassified_entities", report_unclassified_entities)
    workflow.add_node("process_entities", process_entities)
    workflow.add_node("log_process", log_process)
    workflow.add_node("handle_error", handle_error)

    # Connect the workflow
    workflow.add_edge("setup_working_directory", "extract_document_pages")
    workflow.add_edge("extract_document_pages", "detect_entities")
    workflow.add_edge("detect_entities", "classify_entities")
    workflow.add_edge("classify_entities", "report_unclassified_entities")

    # Add conditional edge from report_unclassified to either process_entities or wait for user input
    workflow.add_conditional_edges(
        "report_unclassified_entities",
        decide_if_user_input_needed,
        {
            "continue_processing": "process_entities",
            "need_user_input": END,  # Wait for user input
            "error": "handle_error",
        },
    )

    workflow.add_edge("process_entities", "log_process")
    workflow.add_edge("log_process", END)
    workflow.add_edge("handle_error", END)

    # Set the entry point
    workflow.set_entry_point("setup_working_directory")

    return workflow


# Function to continue workflow after user input
def continue_with_user_input(graph, state, user_classifications):
    """
    Continue the workflow with user input for unclassified entities.
    """
    # Update state with user classifications
    updated_state = process_user_input(state, user_classifications)

    # Continue workflow from process_entities
    return graph.continue_from_node("process_entities", updated_state)


# Example usage
def run_document_processing(document_id):
    """
    Run the document processing pipeline.
    """
    # Initialize state with empty lists/dicts
    state = PipelineState(
        document_id=document_id,
        working_dir="",
        images=[],
        entities={},
        unclassified_entities={},
        logs=[],
        error=None
    )

    # Create and compile the workflow
    workflow = create_document_processing_graph()
    app = workflow.compile()

    # Run the workflow
    result = app.invoke(state)
    
    # Print result details
    print("\nProcessing Results:")
    print("==================")
    if isinstance(result, dict):
        for key, value in result.items():
            if key == 'logs' and value:
                print(f"\nProcessing Logs:")
                for log in value:
                    print(f"- {log.get('step', 'unknown')}: {log.get('message', '')}")
            else:
                print(f"\n{key}:")
                print(f"{value}")

    return result


# Main entrypoint for the application
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <document_id>")
        sys.exit(1)

    document_id = sys.argv[1]
    result = run_document_processing(document_id)

    print("\nPipeline completed.")
    if isinstance(result, dict):
        print(f"Processed {len(result.get('entities', {}))} entities")
    print(f"Output available in {document_id}_working/output")
