using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace PostgreSQLRepository.Migrations
{
    /// <inheritdoc />
    public partial class AddApprovedAndFramesCountToErrorReport : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<bool>(
                name: "Approved",
                table: "ErrorReports",
                type: "boolean",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<int>(
                name: "FramesCount",
                table: "ErrorReports",
                type: "integer",
                nullable: false,
                defaultValue: 0);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "Approved",
                table: "ErrorReports");

            migrationBuilder.DropColumn(
                name: "FramesCount",
                table: "ErrorReports");
        }
    }
}
