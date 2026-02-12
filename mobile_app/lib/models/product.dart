class Product {
  final int id;
  final String name;
  final String? manufacturer;
  final String? barcode;

  Product({required this.id, required this.name, this.manufacturer, this.barcode});

  factory Product.fromJson(Map<String, dynamic> json) => Product(
        id: json['id'] as int,
        name: (json['name'] ?? '').toString(),
        manufacturer: json['manufacturer']?.toString(),
        barcode: json['barcode']?.toString(),
      );
}
