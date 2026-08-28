# CS 180 GitHub Pages Site

Static portfolio site for CS 180 project writeups.

## Local preview

Open `index.html` in a browser, or run a tiny local server:

```sh
python3 -m http.server 8000
```

Then visit `http://localhost:8000`.

## Publish to GitHub Pages

1. Create a new repository on GitHub.
2. Push this folder to the repository's `main` branch.
3. In GitHub, open `Settings` -> `Pages`.
4. Under `Build and deployment`, choose `GitHub Actions`.
5. Push to `main`; the workflow in `.github/workflows/pages.yml` will deploy the site.

Your published URL will usually be:

```text
https://YOUR-USERNAME.github.io/YOUR-REPOSITORY/
```

For a user site, name the repository `YOUR-USERNAME.github.io`.
