import 'package:flutter/material.dart';

class ErrorStubScreen extends StatelessWidget {
  const ErrorStubScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Сообщить об ошибке')),
      body: const Center(
        child: Text('Пока заглушка. Позже добавим форму и загрузку видео.'),
      ),
    );
  }
}
