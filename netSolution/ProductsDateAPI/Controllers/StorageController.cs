using Contracts;
using Contracts.Dtos;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
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

    public StorageController(StoredProductRepository storageRepo, IRepository<StoredProduct> repo)
    {
        _storageRepo = storageRepo;
        _repo = repo;
    }

    [HttpGet]
    public async Task<ActionResult<List<StoredProduct>>> GetMy(CancellationToken ct)
    {
        var userId = User.GetUserIdOrThrow();
        return Ok(await _storageRepo.GetByUserAsync(userId, ct));
    }

    [HttpPost]
    public async Task<ActionResult<StoredProduct>> Add(StoredProductCreateRequest req, CancellationToken ct)
    {
        var userId = User.GetUserIdOrThrow();

        var sp = new StoredProduct
        {
            UserId = userId,
            ProductId = req.ProductId,
            ManufactureAt = req.ManufactureAt,
            ExpiryAt = req.ExpiryAt,
            CreatedAt = DateTime.UtcNow
        };

        await _repo.AddAsync(sp, ct);
        return Ok(sp);
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
}
