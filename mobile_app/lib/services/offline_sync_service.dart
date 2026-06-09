import '../api/api_client.dart';
import '../auth/auth_state.dart';
import 'local_storage_repository.dart';
import 'pending_report_repository.dart';

class OfflineSyncService {
  final LocalStoredProductRepository products;
  final PendingReportRepository reports;

  OfflineSyncService({
    LocalStoredProductRepository? products,
    PendingReportRepository? reports,
  })  : products = products ?? LocalStoredProductRepository(),
        reports = reports ?? PendingReportRepository();

  Future<void> trySync(AuthState auth) async {
    if (!auth.isAuthed) return;
    final api = ApiClient(token: auth.token);

    try {
      await products.syncPendingAdds(api);
    } catch (_) {}

    try {
      await reports.syncPendingReports(api);
    } catch (_) {}
  }
}
