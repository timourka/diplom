using Microsoft.EntityFrameworkCore;
using Models.Entities;

namespace PostgreSQLRepository;

public class StoredProductRepository : Repository<StoredProduct>
{
    public StoredProductRepository(AppDbContext db) : base(db) { }

    public async Task<List<StoredProduct>> GetByUserAsync(int userId, CancellationToken ct = default)
    {
        return await Db.StoredProducts
            .AsNoTracking()
            .Include(x => x.Product)
            .Where(x => x.UserId == userId)
            .OrderByDescending(x => x.CreatedAt)
            .ToListAsync(ct);
    }
}
