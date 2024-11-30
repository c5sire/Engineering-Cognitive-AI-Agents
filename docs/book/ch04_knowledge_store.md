# Design Document: `KnowledgeStorage`

## Overview

The `KnowledgeStorage` class provides a simple yet scalable architecture for storing, retrieving, and managing structured knowledge entries in a file-based system. This implementation is particularly suitable for lightweight applications that require persistent storage without relying on external databases. Each entry, represented by the `Knowledge` model, is stored as a JSON file, enabling portability and easy integration with file systems.

For validation and structure, the Python library **Pydantic** is used to ensure type-safety and consistent input validation for all operations.

---

## Goals

- **Simple Storage Mechanism**: Use files on the local filesystem for lightweight use cases without database dependencies.
- **CRUD Functionality**: Provide foundational Create, Read, Update, and Delete (CRUD) operations for managing knowledge entries.
- **Metadata Support**: Enable contextual metadata for each knowledge entry.
- **Scalability (within limits)**: Design for small-to-medium data volumes where file-based storage is appropriate.

---

## Scope and Features

The current implementation includes the following features:

1. **Knowledge Representation**:
   Using the `Knowledge` class, every knowledge entry contains:

   - A unique identifier (`id`).
   - The main content/text (`content`).
   - Metadata attached as a dictionary (`context`).
   - Timestamps for creation and last update (`created_at`, `updated_at`).

2. **File-based Storage**:

   - Entries are persisted as JSON files.
   - Each knowledge entry's filename is generated using its unique ID: `{id}.json`.

3. **CRUD APIs**:

   - **Create**: Add new knowledge entries to the storage.
   - **Read**: Fetch a specific knowledge entry by its ID.
   - **Update**: Update the content and/or metadata of an entry.
   - **List All**: Retrieve all stored entries for quick visibility.

4. **Validation**:
   The implementation leverages **Pydantic** for data validation, ensuring:

   - Proper typing for every entry field.
   - Input consistency for all APIs.

5. **Asynchronous Operations**:
   All interactions, including file I/O, are asynchronous, making the system ready for integration with event loops (e.g., Python's `asyncio`).

---

## Design Details

### Architecture

1. **Storage Directory**:
   The `KnowledgeStorage` class requires a `Path` to a storage directory. This directory is created automatically (if it doesn't exist), ensuring seamless initialization during setup.

2. **Filename and IDs**:

   - Unique IDs are generated using `uuid4()`.
   - Each knowledge entry maps to a unique JSON file in the storage directory.

3. **File Operations**:

   - JSON files are used as the underlying storage format for simplicity and compatibility.
   - Operations such as `store`, `load`, `update`, and `list_all` involve reading/writing to these JSON files.

4. **Timestamps**:
   - Both `created_at` and `updated_at` fields are automatically managed for transparency in data update history.

### Operations

#### `store`

- **Purpose**: Add a new knowledge entry to the storage.
- **Input**: Content (`content`) and metadata (`context`).
- **Output**: Returns a unique ID for the stored entry.
- **Behavior**:
  - Generates a UUID.
  - Constructs a `Knowledge` object.
  - Dumps the object to a JSON file.

#### `load`

- **Purpose**: Retrieve a specific knowledge item by ID.
- **Input**: Knowledge ID.
- **Output**: Returns a `Knowledge` object.
- **Behavior**:
  - Loads a file with the corresponding filename.
  - Validates the contents against the `Knowledge` schema.

#### `update`

- **Purpose**: Modify existing knowledge content or metadata.
- **Input**: Knowledge ID, optional new content (`content`), and metadata updates (`context`).
- **Output**: Returns an updated `Knowledge` object.
- **Behavior**:
  - Fetches the existing knowledge file.
  - Applies updates to the content and/or metadata fields.
  - Updates the `updated_at` timestamp in the object.

#### `list_all`

- **Purpose**: List all stored knowledge entries.
- **Output**: Returns a list of `Knowledge` objects.
- **Behavior**:
  - Reads all JSON files in the storage directory.
  - Instantiates `Knowledge` objects for each entry.

---

## Code Standards

- **Pydantic Validation**: Ensures all inputs and outputs conform to the specified schema.
- **Asyncio Ready**: The code is asynchronous, which ensures compatibility with modern Python frameworks and scalability during I/O-heavy operations.
- **Type Hints**: Uses Python's type hints, ensuring IDE support and code maintainability.
- **Modularity**: Classes (`Knowledge` and `KnowledgeStorage`) are self-contained, allowing reusability and testability.

---

## Test Plan

### Unit Tests

All functionalities are covered by the following test cases, implemented in `tests/test_storage.py`:

1. **Test Storing Knowledge**:

   - Validate saving a new entry.
   - Ensure that the returned ID is unique and valid.

2. **Test Loading**:

   - Ensure an entry previously stored can be retrieved.
   - Verify that the content and metadata match the input.

3. **Test Updating**:

   - Validate updating both content and metadata.
   - Ensure that `updated_at` reflects the modification.

4. **Test Listing**:

   - Save multiple entries in storage.
   - Verify that the `list_all` method retrieves all entries.
   - Confirm that all stored data matches input values.

5. **Edge Cases**:
   - Attempt to load/update an invalid or missing ID (expect `FileNotFoundError`).
   - Update with `None` for optional fields (expect no changes to those fields).

---

## Tradeoffs and Future Considerations

### Tradeoffs

1. **File-based Storage vs Database**:

   - Simplicity: File-based storage is easy to implement and requires no external dependencies.
   - Limitation: As data grows, file I/O may become slow, and retrieval operations might not scale effectively.

2. **Timestamps**:

   - Only basic management (created and updated) is currently included. More complicated scenarios (e.g., versioning or audit history) are not covered.

3. **Asynchronous Behavior**:
   - While asynchronous operations enable integration with event loops, true concurrency benefits depend on how the storage is deployed.

### Future Features

1. **Database Integration**:

   - Transition to a database-backed system (e.g., SQLite, Postgres) as the number of stored entries scales beyond manageable file counts.

2. **Knowledge Deletion**:

   - Add a `delete` method for removing knowledge entries.

3. **Indexing and Search**:

   - Incorporate indexing to provide efficient keyword-based searches across stored data.

4. **Versioning / History**:

   - Track historical changes to knowledge entries for better traceability.

5. **Advanced Metadata Management**:

   - Extend context metadata to support nested structures or larger objects (e.g., linked files, references).

6. **Concurrency Safety**:
   - Ensure operations (e.g., `update`) are safe in multi-user or multi-process environments.

---

## Summary

The `KnowledgeStorage` implementation provides a foundation for storing and managing structured knowledge entries in file-based systems. It accomplishes CRUD operations efficiently and lays the groundwork for extension into more complex systems. Planned enhancements like database support and advanced metadata handling will further increase its capabilities.
