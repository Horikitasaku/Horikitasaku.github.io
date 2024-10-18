const themeDir = __dirname + "/../../";

module.exports = {
  plugins: [
    require("tailwindcss")(themeDir + "assets/css/tailwind.config.js"),
    require("postcss-import")(),
    require("postcss-nested")(),
    require("postcss-preset-env")({
      stage: 1,
      features: {
        'nesting-rules': true,
      },
    }),
    require("autoprefixer")({
      path: [themeDir],
    }),
    require("cssnano")({
      preset: "default",
    }),
  ],
};
