using System;
using Microsoft.EntityFrameworkCore.Migrations;
using Npgsql.EntityFrameworkCore.PostgreSQL.Metadata;

#nullable disable

namespace PostgreSQLRepository.Migrations
{
    /// <inheritdoc />
    public partial class AddPullTrainingAndModelPublishing : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<DateTime>(
                name: "DeletedAt",
                table: "ModelVersions",
                type: "timestamp with time zone",
                nullable: true);

            migrationBuilder.AddColumn<bool>(
                name: "IsDeleted",
                table: "ModelVersions",
                type: "boolean",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<bool>(
                name: "IsPinned",
                table: "ModelVersions",
                type: "boolean",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<bool>(
                name: "IsPublished",
                table: "ModelVersions",
                type: "boolean",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<string>(
                name: "MobileModelContentType",
                table: "ModelVersions",
                type: "character varying(128)",
                maxLength: 128,
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "MobileModelFileName",
                table: "ModelVersions",
                type: "character varying(512)",
                maxLength: 512,
                nullable: true);

            migrationBuilder.CreateTable(
                name: "TrainingJobs",
                columns: table => new
                {
                    Id = table.Column<int>(type: "integer", nullable: false)
                        .Annotation("Npgsql:ValueGenerationStrategy", NpgsqlValueGenerationStrategy.IdentityByDefaultColumn),
                    JobId = table.Column<string>(type: "character varying(128)", maxLength: 128, nullable: false),
                    Status = table.Column<string>(type: "character varying(32)", maxLength: 32, nullable: false),
                    Message = table.Column<string>(type: "text", nullable: true),
                    CreatedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: false),
                    StartedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: true),
                    FinishedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: true),
                    AssignedAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: true),
                    HeartbeatAt = table.Column<DateTime>(type: "timestamp with time zone", nullable: true),
                    ImagesCount = table.Column<int>(type: "integer", nullable: false),
                    ClientId = table.Column<string>(type: "character varying(256)", maxLength: 256, nullable: true),
                    DatasetZipPath = table.Column<string>(type: "character varying(2048)", maxLength: 2048, nullable: false),
                    BaseModel = table.Column<string>(type: "character varying(256)", maxLength: 256, nullable: true),
                    Epochs = table.Column<int>(type: "integer", nullable: false),
                    ImgSize = table.Column<int>(type: "integer", nullable: false),
                    Batch = table.Column<int>(type: "integer", nullable: false),
                    Device = table.Column<string>(type: "character varying(64)", maxLength: 64, nullable: true),
                    ExportInt8 = table.Column<bool>(type: "boolean", nullable: false),
                    ExportNms = table.Column<bool>(type: "boolean", nullable: false),
                    MobileFormat = table.Column<string>(type: "character varying(32)", maxLength: 32, nullable: true),
                    QuantizationFraction = table.Column<double>(type: "double precision", nullable: false),
                    BestWeightsPath = table.Column<string>(type: "character varying(2048)", maxLength: 2048, nullable: true),
                    MobileModelPath = table.Column<string>(type: "character varying(2048)", maxLength: 2048, nullable: true),
                    MobileModelFileName = table.Column<string>(type: "character varying(512)", maxLength: 512, nullable: true),
                    MobileModelContentType = table.Column<string>(type: "character varying(128)", maxLength: 128, nullable: true),
                    MetricsJson = table.Column<string>(type: "text", nullable: true),
                    CancellationRequested = table.Column<bool>(type: "boolean", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_TrainingJobs", x => x.Id);
                });

            migrationBuilder.CreateIndex(
                name: "IX_ModelVersions_IsDeleted",
                table: "ModelVersions",
                column: "IsDeleted");

            migrationBuilder.CreateIndex(
                name: "IX_ModelVersions_IsPublished",
                table: "ModelVersions",
                column: "IsPublished");

            migrationBuilder.CreateIndex(
                name: "IX_TrainingJobs_JobId",
                table: "TrainingJobs",
                column: "JobId",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_TrainingJobs_Status",
                table: "TrainingJobs",
                column: "Status");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "TrainingJobs");

            migrationBuilder.DropIndex(
                name: "IX_ModelVersions_IsDeleted",
                table: "ModelVersions");

            migrationBuilder.DropIndex(
                name: "IX_ModelVersions_IsPublished",
                table: "ModelVersions");

            migrationBuilder.DropColumn(
                name: "DeletedAt",
                table: "ModelVersions");

            migrationBuilder.DropColumn(
                name: "IsDeleted",
                table: "ModelVersions");

            migrationBuilder.DropColumn(
                name: "IsPinned",
                table: "ModelVersions");

            migrationBuilder.DropColumn(
                name: "IsPublished",
                table: "ModelVersions");

            migrationBuilder.DropColumn(
                name: "MobileModelContentType",
                table: "ModelVersions");

            migrationBuilder.DropColumn(
                name: "MobileModelFileName",
                table: "ModelVersions");
        }
    }
}
