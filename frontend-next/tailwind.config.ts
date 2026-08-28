import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "#0b0f17",
        panel: "#121826",
        panel2: "#1a2234",
        border: "#243049",
        accent: "#5b8cff",
        muted: "#8b97ad",
        good: "#37d399",
        warn: "#f5b74e",
        bad: "#ff6b6b",
      },
    },
  },
  plugins: [],
};
export default config;
