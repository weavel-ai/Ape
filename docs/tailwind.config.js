/** @type {import('tailwindcss').Config} */

module.exports = {
  darkMode: "class",
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/**/*.{js,jsx,ts,tsx,mdx}",
    "./theme.config.tsx",
    "./lib/**/*.{js,jsx,ts,tsx,mdx}",
  ],
  theme: {
    maxWidth: {
      "8xl": "88rem",
    },
  },
};
