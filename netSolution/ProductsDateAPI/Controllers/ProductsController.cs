using Contracts;
using Contracts.Dtos;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Models.Entities;

namespace ProductsDateAPI.Controllers;

[ApiController]
[Route("api/[controller]")]
public class ProductsController : ControllerBase
{
    private readonly IRepository<Product> _repo;

    public ProductsController(IRepository<Product> repo) => _repo = repo;

    [HttpGet]
    public async Task<ActionResult<List<Product>>> GetAll(CancellationToken ct)
        => Ok(await _repo.GetAllAsync(ct));

    [HttpGet("{id:int}")]
    public async Task<ActionResult<Product>> Get(int id, CancellationToken ct)
    {
        var p = await _repo.GetByIdAsync(id, ct);
        return p is null ? NotFound() : Ok(p);
    }

    // Пользовательский сценарий добавления товара работает через /api/storage:
    // приложение передаёт название и дату, а сервер находит существующий Product или создаёт новый.
    // Поэтому создание Product не является админской возможностью.
    [Authorize]
    [HttpPost]
    public async Task<ActionResult<Product>> Create(ProductCreateRequest req, CancellationToken ct)
    {
        var p = new Product
        {
            Name = req.Name,
            Manufacturer = req.Manufacturer,
            Barcode = req.Barcode
        };

        await _repo.AddAsync(p, ct);
        return CreatedAtAction(nameof(Get), new { id = p.Id }, p);
    }

    // Редактирование справочника уже является административным действием.
    [Authorize(Policy = "AdminOnly")]
    [HttpPut("{id:int}")]
    public async Task<ActionResult> Update(int id, ProductUpdateRequest req, CancellationToken ct)
    {
        var p = await _repo.GetByIdAsync(id, ct);
        if (p is null) return NotFound();

        p.Name = req.Name;
        p.Manufacturer = req.Manufacturer;
        p.Barcode = req.Barcode;

        await _repo.UpdateAsync(p, ct);
        return NoContent();
    }

    [Authorize(Policy = "AdminOnly")]
    [HttpDelete("{id:int}")]
    public async Task<ActionResult> Delete(int id, CancellationToken ct)
    {
        await _repo.DeleteAsync(id, ct);
        return NoContent();
    }
}
