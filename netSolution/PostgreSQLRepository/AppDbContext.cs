using Microsoft.EntityFrameworkCore;
using Models.Entities;

namespace PostgreSQLRepository;

public class AppDbContext : DbContext
{
    public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }

    public DbSet<User> Users => Set<User>();
    public DbSet<Product> Products => Set<Product>();
    public DbSet<StoredProduct> StoredProducts => Set<StoredProduct>();
    public DbSet<VideoSample> VideoSamples => Set<VideoSample>();
    public DbSet<ModelVersion> ModelVersions => Set<ModelVersion>();
    public DbSet<ErrorReport> ErrorReports => Set<ErrorReport>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        base.OnModelCreating(modelBuilder);

        modelBuilder.Entity<User>()
            .HasIndex(x => x.Email)
            .IsUnique();

        modelBuilder.Entity<Product>()
            .HasIndex(x => x.Barcode);

        modelBuilder.Entity<StoredProduct>()
            .HasOne(x => x.User)
            .WithMany(x => x.StoredProducts)
            .HasForeignKey(x => x.UserId)
            .OnDelete(DeleteBehavior.Cascade);

        modelBuilder.Entity<StoredProduct>()
            .HasOne(x => x.Product)
            .WithMany(x => x.StoredProducts)
            .HasForeignKey(x => x.ProductId)
            .OnDelete(DeleteBehavior.Restrict);

        modelBuilder.Entity<ErrorReport>()
            .HasOne(x => x.User)
            .WithMany(x => x.ErrorReports)
            .HasForeignKey(x => x.UserId)
            .OnDelete(DeleteBehavior.Cascade);

        modelBuilder.Entity<ErrorReport>()
            .HasOne(x => x.Video)
            .WithMany(x => x.ErrorReports)
            .HasForeignKey(x => x.VideoId)
            .OnDelete(DeleteBehavior.SetNull);

        modelBuilder.Entity<ErrorReport>()
            .HasOne(x => x.ModelVersion)
            .WithMany(x => x.ErrorReports)
            .HasForeignKey(x => x.ModelVersionId)
            .OnDelete(DeleteBehavior.SetNull);
    }
}
