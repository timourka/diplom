namespace AdminApp;

internal static class Program
{
    [STAThread]
    static void Main()
    {
        ApplicationConfiguration.Initialize();

        while (true)
        {
            using var loginForm = new FormLogin();
            if (loginForm.ShowDialog() != DialogResult.OK || string.IsNullOrWhiteSpace(loginForm.Token))
                return;

            using var mainForm = new FormMain(loginForm.Token);
            Application.Run(mainForm);

            if (!mainForm.LoggedOut)
                return;
        }
    }
}
