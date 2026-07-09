# Testing Local Files Locally

Use this when you want to test real files against the local `/v1/parse` API.

## 1. Add Local Environment Values

Create or update `apps/api/.env`:

```env
CORE_OPENROUTER_API_KEY=your-openrouter-key
CORE_DEFAULT_MODEL=openai/gpt-5.2
CORE_DEFAULT_SYSTEM_PROMPT=Return factual JSON only.
CORE_TIMEOUT_SECONDS=60
CORE_MAX_UPLOAD_BYTES=25000000
CORE_API_CONSUMERS='[{"id":"local","api_key":"local-test-key","revoked":false}]'
```

For `.doc` files, LibreOffice must be available. If it is not on your `PATH`, add:

```env
CORE_LIBREOFFICE_BINARY=/absolute/path/to/soffice
```

## 2. Start The Local API

```bash
just dev
```

The API should be available at:

```text
http://127.0.0.1:8000
```

## 3. Test One File With Curl

Change the file path at the end of the command.

```bash
curl -X POST http://127.0.0.1:8000/v1/parse \
  -H "X-API-Key: local-test-key" \
  -F 'schema={
    "type":"object",
    "properties":{
      "title":{"type":["string","null"]},
      "summary":{"type":["string","null"]}
    },
    "required":["title","summary"],
    "additionalProperties":false
  }' \
  -F "file=@/absolute/path/to/your-file.pdf"
```

## 4. Test Different File Types

Use the same command and only change the file path:

```bash
-F "file=@/absolute/path/to/sample.pdf"
-F "file=@/absolute/path/to/sample.png"
-F "file=@/absolute/path/to/sample.jpg"
-F "file=@/absolute/path/to/sample.docx"
-F "file=@/absolute/path/to/sample.doc"
-F "file=@/absolute/path/to/sample.html"
-F "file=@/absolute/path/to/sample.xlsx"
-F "file=@/absolute/path/to/sample.xls"
```

Supported file types:

```text
PDF, scanned PDF, PNG, JPG/JPEG, DOCX, DOC, HTML/HTM, XLSX, XLS
```

For scanned PDFs and images, use a vision-capable OpenRouter model.

## 5. Expected Response Shape

Successful responses look like this:

```json
{
  "data": {
    "title": "...",
    "summary": "..."
  },
  "meta": {
    "documentType": "pdf",
    "extractionMode": "native",
    "pageCount": 1,
    "ocrPageCount": 0,
    "converter": "markitdown"
  }
}
```

If something fails, check the stable error code:

```json
{
  "error": {
    "code": "unsupported_file_type",
    "message": "Unsupported file type"
  }
}
```
