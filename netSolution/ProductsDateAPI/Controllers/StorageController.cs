using Contracts;
using Contracts.Dtos;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Models.Entities;
using PostgreSQLRepository;
using ProductsDateAPI.Helpers;

namespace ProductsDateAPI.Controllers;

[ApiController]
[Route("api/[controller]")]
[Authorize]
public class StorageController : ControllerBase
{
    private readonly StoredProductRepository _storageRepo;
    private readonly IRepository<StoredProduct> _repo;
    private readonly AppDbContext _db;

    public StorageController(
        StoredProductRepository storageRepo,
        IRepository<StoredProduct> repo,
        AppDbContext db)
    {
        _storageRepo = storageRepo;
        _repo = repo;
        _db = db;
    }

    [HttpGet]
    public async Task<ActionResult<List<StoredProductDto>>> GetMy(CancellationToken ct)
    {
        var userId = User.GetUserIdOrThrow();
        var items = await _storageRepo.GetByUserAsync(userId, ct);
        return Ok(items.Select(MapStoredProduct).ToList());
    }

    [HttpPost]
    public async Task<ActionResult<StoredProductDto>> Add(StoredProductCreateRequest req, CancellationToken ct)
    {
        var userId = User.GetUserIdOrThrow();

        var product = await ResolveProductAsync(req, ct);
        if (product is null)
        {
            return BadRequest("Нужно передать productId или непустое название продукта.");
        }

        var sp = new StoredProduct
        {
            UserId = userId,
            ProductId = product.Id,
            ManufactureAt = NormalizeToUtc(req.ManufactureAt),
            ExpiryAt = NormalizeToUtc(req.ExpiryAt),
            CreatedAt = DateTime.UtcNow
        };

        await _repo.AddAsync(sp, ct);

        sp.Product = product;
        return Ok(MapStoredProduct(sp));
    }

    [HttpDelete("{id:int}")]
    public async Task<ActionResult> Delete(int id, CancellationToken ct)
    {
        // Защита: удалять можно только своё
        var userId = User.GetUserIdOrThrow();
        var sp = await _repo.GetByIdAsync(id, ct);
        if (sp is null) return NotFound();
        if (sp.UserId != userId) return Forbid();

        await _repo.DeleteAsync(id, ct);
        return NoContent();
    }


    private static StoredProductDto MapStoredProduct(StoredProduct entity)
    {
        return new StoredProductDto
        {
            Id = entity.Id,
            UserId = entity.UserId,
            ProductId = entity.ProductId,
            ManufactureAt = entity.ManufactureAt,
            ExpiryAt = entity.ExpiryAt,
            CreatedAt = entity.CreatedAt,
            Product = entity.Product is null
                ? null
                : new ProductSummaryDto
                {
                    Id = entity.Product.Id,
                    Name = entity.Product.Name,
                    Manufacturer = entity.Product.Manufacturer,
                    Barcode = entity.Product.Barcode
                }
        };
    }

    private async Task<Product?> ResolveProductAsync(StoredProductCreateRequest req, CancellationToken ct)
    {
        if (req.ProductId is int productId)
        {
            return await _db.Products.FirstOrDefaultAsync(x => x.Id == productId, ct);
        }

        var name = req.ProductName?.Trim();
        if (string.IsNullOrWhiteSpace(name))
        {
            return null;
        }

        var existing = await _db.Products
            .FirstOrDefaultAsync(x => x.Name.ToLower() == name.ToLower(), ct);

        if (existing is not null)
        {
            return existing;
        }

        var created = new Product
        {
            Name = name
        };

        _db.Products.Add(created);
        await _db.SaveChangesAsync(ct);
        return created;
    }
    private static DateTime? NormalizeToUtc(DateTime? value)
    {
        if (value is null)
        {
            return null;
        }

        var dt = value.Value;
        return dt.Kind switch
        {
            DateTimeKind.Utc => dt,
            DateTimeKind.Local => dt.ToUniversalTime(),
            _ => DateTime.SpecifyKind(dt, DateTimeKind.Utc)
        };
    }

}
