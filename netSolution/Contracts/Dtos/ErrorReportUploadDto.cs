namespace Contracts.Dtos;

public record UploadDatasetResponse(int ReportId, int VideoSampleId, string Folder);

public record ReportValidationFrameResponse(
    string ValidationToken,
    string FileName,
    string ImageBase64,
    string ContentType
);
