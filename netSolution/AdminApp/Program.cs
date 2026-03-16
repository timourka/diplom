namespace AdminApp;

internal static class Program
{
    [STAThread]
    static void Main()
    {
        ApplicationConfiguration.Initialize();

        using var loginForm = new FormLogin();
        if (loginForm.ShowDialog() == DialogResult.OK && !string.IsNullOrWhiteSpace(loginForm.Token))
        {
            Application.Run(new FormMain(loginForm.Token));
        }
    }
}