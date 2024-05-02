"""
Friendlier Document AI OCR.

These helper functions are designed to be more user friendly and robustly documented than the official SDK.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import google.auth
from google.api_core import operation
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import InternalServerError, RetryError
from google.cloud import documentai, storage
from google.cloud.documentai_toolbox import gcs_utilities
from pypdf import PdfReader

# Setup logging
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

# Synchronous/Online Processing Limits
MAX_SYNCRONOUS_PAGES = 15
MAX_SYNCRONOUS_SIZE = 20  # in MB
MAX_SYNCRONOUS_REQUESTS = 120  # per processor, per minute

# Async/Batch Limits
MAX_BATCH_PAGES = 500
MAX_BATCH_SIZE = 1000  # number of files
MAX_BATCH_REQUESTS = (
    10  # max request limit as of v1beta2 (client should be using v1 now)
)

TIMEOUT = 60 * 10  # 10 minutes


@dataclass
class DocumentAIConfig:
    project_id: str = ""
    location: str = "us"  # Format is "us" or "eu"
    processor_id: str = ""  # Requirement: Create processor before running sample
    # Refer to https://cloud.google.com/document-ai/docs/manage-processor-versions for more information
    processor_version: str = "rc"
    file_path: str = ""  # This is not used when doing batch processing
    # Refer to https://cloud.google.com/document-ai/docs/file-types for supported file types
    mime_type: str = "application/pdf"


@dataclass
class BatchConfig:
    gcs_bucket_name: str
    gcs_prefix: str  # "Subdirectory" to batch OCR process
    gcs_output_uri: str  # Output URI for batch OCR results
    batch_size: int = 50
    field_mask: Optional[str] = (
        None  # Optional. The fields to return in the Document object.
    )


@dataclass
class BlobStorageTransferConfig:
    project_id: str
    source_directory: str  # Local directory where the files come from
    gcs_bucket_name: str
    gcs_prefix: str = ""
    max_workers: int = 8


def build_doc_processor_client(
    location: str,
) -> documentai.DocumentProcessorServiceClient:
    # You must set the `api_endpoint` if you use a location other than "us".
    # According to best practice reuse client when possible
    # https://cloud.google.com/apis/docs/client-libraries-best-practices
    credentials, project = google.auth.default()
    logger.info(
        f"Using project credentials for the following gcloud auth default: {project}"
    )
    client = documentai.DocumentProcessorServiceClient(
        credentials=credentials,
        client_options=ClientOptions(
            api_endpoint=f"{doc_ai_config.location}-documentai.googleapis.com"
        ),
    )
    return client


def process_document_ocr(
    client: documentai.DocumentProcessorServiceClient,
    doc_ai_config: DocumentAIConfig,
    process_options: Optional[documentai.ProcessOptions] = None,
) -> documentai.Document:
    """
    Process a single, local document using the Document AI OCR processor.
    """
    if doc_ai_config.mime_type == "application/pdf":
        with PdfReader(doc_ai_config.file_path) as reader:
            if len(reader.pages) > MAX_SYNCRONOUS_PAGES:
                raise ValueError(
                    f"Too many pages to process, max is {MAX_SYNCRONOUS_PAGES}"
                )

    # The full resource name of the processor version, e.g.:
    # `projects/{project_id}/locations/{location}/processors/{processor_id}/processorVersions/{processor_version_id}`
    # You must create a processor before running this sample.
    name = client.processor_version_path(
        doc_ai_config.project_id,
        doc_ai_config.location,
        doc_ai_config.processor_id,
        doc_ai_config.processor_version,
    )

    # Read the file into memory
    with open(doc_ai_config.file_path, "rb") as image:
        image_content = image.read()

    # Configure the process request
    request = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(
            content=image_content, mime_type=doc_ai_config.mime_type
        ),
        # Only for Document OCR processor
        process_options=process_options,
    )

    result = client.process_document(request=request)

    # For a full list of `Document` object attributes, reference this page:
    # https://cloud.google.com/document-ai/docs/reference/rest/v1/Document
    return result.document


def batch_process_document_ocr(
    batch_config: BatchConfig,
    client: documentai.DocumentProcessorServiceClient,
    doc_ai_config: DocumentAIConfig,
    process_options: Optional[documentai.ProcessOptions] = None,
    async_operation: bool = False,
) -> List[operation.Operation]:
    # Setup where to write results
    output_config = documentai.DocumentOutputConfig(
        gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(
            gcs_uri=batch_config.gcs_output_uri, field_mask=batch_config.field_mask
        )
    )
    batches = gcs_utilities.create_batches(
        gcs_bucket_name=batch_config.gcs_bucket_name,
        gcs_prefix=batch_config.gcs_prefix,
        batch_size=batch_config.batch_size,
    )
    if len(batches) > MAX_BATCH_REQUESTS and async_operation:
        raise ValueError(
            f"Currently async is not supported exceeding max batch requests of {MAX_BATCH_REQUESTS}, "
            f"we have {len(batches)}"
        )

    # The full resource name of the processor version, e.g.:
    # `projects/{project_id}/locations/{location}/processors/{processor_id}/processorVersions/{processor_version_id}`
    # You must create a processor before running this sample.
    name = client.processor_version_path(
        doc_ai_config.project_id,
        doc_ai_config.location,
        doc_ai_config.processor_id,
        doc_ai_config.processor_version,
    )

    batch_operations = []
    for batch in batches:
        # Configure the process request
        request = documentai.BatchProcessRequest(
            name=name,
            input_documents=batch,
            document_output_config=output_config,
            # Only for Document OCR processor
            process_options=process_options,
        )

        operation = client.batch_process_documents(request=request)
        batch_operations.append(operation)

        # TODO: split out batch operation so that we can run async using mp and keep track of total requests
        if async_operation:
            continue
        try:
            logger.info(
                f"Waiting for operation {operation.operation.name} to complete..."
            )
            total_wait_time = 0
            while not operation.done():
                time.sleep(15)
                total_wait_time += 15
                logger.info(f"{total_wait_time} seconds elapsed")
                if total_wait_time > TIMEOUT:
                    logger.info(f"Operation {operation.operation.name} timed out.")
                    break
        except (RetryError, InternalServerError) as e:
            logger.error(e.message)

    return batch_operations


def check_batch_operation_status(batch_operation: operation.Operation) -> bool:
    metadata = documentai.BatchProcessMetadata(batch_operation.metadata)
    if metadata.state != documentai.BatchProcessMetadata.State.SUCCEEDED:
        operation_output_path = Path(
            metadata.individual_process_statuses[0]["output_gcs_destination"]
        )
        operation_id = operation_output_path.parents[0].stem
        logger.error(
            f"Batch operation {operation_id} failed.\nOutput path: {operation_output_path}"
        )
        return False
    return True


def upload_files_to_blob_storage(
    storage_client: storage.Client,
    blob_transfer_config: BlobStorageTransferConfig,
) -> List[operation.Operation]:
    bucket = storage_client.bucket(blob_transfer_config.gcs_bucket_name)
    filepaths = get_filepaths_from_directory(blob_transfer_config.source_directory)

    results = storage.transfer_manager.upload_many_from_filenames(
        bucket,
        filepaths,
        source_directory=blob_transfer_config.source_directory,
        max_workers=blob_transfer_config.max_workers,
        blob_name_prefix=blob_transfer_config.gcs_prefix,
    )
    for name, result in zip(filepaths, results):
        if isinstance(result, Exception):
            logger.error(
                "Failed to upload {} due to exception: {}".format(name, result)
            )
        else:
            logger.info("Uploaded {} to {}.".format(name, bucket.name))


def get_filepaths_from_directory(directory: str) -> List[str]:
    directory_as_path_obj = Path(directory)
    paths = directory_as_path_obj.rglob("*")
    file_paths = [path for path in paths if path.is_file()]
    relative_paths = [path.relative_to(directory) for path in file_paths]
    string_paths = [str(path) for path in relative_paths]
    logger.info("Found {} files.".format(len(string_paths)))
    return string_paths


if __name__ == "__main__":
    FILE_PATH = "test/sample_upload.pdf"
    GCS_BUCKET_NAME = "pointable-ai-staging-doc-ai"
    PROJECT_ID = "PROJECT ID HERE"
    PROCESSOR_ID = "PROCESSOR ID HERE"

    doc_ai_config = DocumentAIConfig(
        project_id=PROJECT_ID,
        location="us",
        processor_id=PROCESSOR_ID,
        processor_version="pretrained-ocr-v2.0-2023-06-02",
        file_path=FILE_PATH,
        mime_type="application/pdf",
    )

    # You can find config options here https://cloud.google.com/document-ai/docs/reference/rest/v1/ProcessOptions#OcrConfig  # noqa: E501
    # There are also premium features here https://cloud.google.com/document-ai/docs/reference/rest/v1/ProcessOptions#premiumfeatures  # noqa: E501
    process_options = documentai.ProcessOptions(
        ocr_config=documentai.OcrConfig(
            enable_image_quality_scores=True,
            enable_symbol=False,
        )
    )
    doc_processor_client = build_doc_processor_client(doc_ai_config.location)

    # # ===Synchronous Processing===
    # document = process_document_ocr(doc_processor_client, doc_ai_config, process_options)
    # logger.info(document.text)
    # # ===End Synchronous Processing===

    # ===Async Processing===
    # Upload files for async processing
    storage_client = storage.Client()
    blob_transfer_config = BlobStorageTransferConfig(
        project_id=PROJECT_ID,
        source_directory="./demo",
        gcs_bucket_name=GCS_BUCKET_NAME,
        gcs_prefix="demo/",
    )
    upload_files_to_blob_storage(storage_client, blob_transfer_config)

    # Setup async batch processing
    OUTPUT_SUBDIR = "demo_output"
    batch_config = BatchConfig(
        gcs_bucket_name=GCS_BUCKET_NAME,
        gcs_prefix="demo/",  # root dir if empty string
        gcs_output_uri=f"gs://{GCS_BUCKET_NAME}/{OUTPUT_SUBDIR}/",
        batch_size=MAX_BATCH_SIZE,
    )

    # If you're getting google.api_core.exceptions.PermissionDenied: 403, you either need
    # to set GOOGLE_APPLICATION_CREDENTIALS to a local credentials.json
    # or run `gcloud auth application-default login` after you set your default project
    batch_operations = batch_process_document_ocr(
        batch_config, doc_processor_client, doc_ai_config, process_options
    )
    for batch_operation in batch_operations:
        check_batch_operation_status(batch_operation)
    # ===End Async Processing===
