using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace PostgreSQLRepository.Migrations
{
    public partial class AddTrainingArtifactsToModelVersion : Migration
    {
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "BaseModel",
                table: "ModelVersions",
                type: "character varying(256)",
                maxLength: 256,
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "BestWeightsPath",
                table: "ModelVersions",
                type: "character varying(2048)",
                maxLength: 2048,
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "ExternalJobId",
                table: "ModelVersions",
                type: "character varying(128)",
                maxLength: 128,
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "MobileFormat",
                table: "ModelVersions",
                type: "character varying(32)",
                maxLength: 32,
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "MobileModelPath",
                table: "ModelVersions",
                type: "character varying(2048)",
                maxLength: 2048,
                nullable: true);

            migrationBuilder.CreateIndex(
                name: "IX_ModelVersions_ExternalJobId",
                table: "ModelVersions",
                column: "ExternalJobId",
                unique: true);
        }

        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropIndex(
                name: "IX_ModelVersions_ExternalJobId",
                table: "ModelVersions");

            migrationBuilder.DropColumn(
                name: "BaseModel",
                table: "ModelVersions");

            migrationBuilder.DropColumn(
                name: "BestWeightsPath",
                table: "ModelVersions");

            migrationBuilder.DropColumn(
                name: "ExternalJobId",
                table: "ModelVersions");

            migrationBuilder.DropColumn(
                name: "MobileFormat",
                table: "ModelVersions");

            migrationBuilder.DropColumn(
                name: "MobileModelPath",
                table: "ModelVersions");
        }
    }
}
