export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#08111d",
        mist: "#d9e8f5",
        tide: "#0c2137",
        pulse: "#6df0c2",
        ember: "#ff9f5a",
        frost: "#91c8ff",
        rose: "#ff7ba5",
      },
      boxShadow: {
        glow: "0 18px 60px rgba(5, 18, 33, 0.28)",
      },
      fontFamily: {
        sans: ["Inter", "sans-serif"],
        display: ["Inter", "sans-serif"],
        body: ["Inter", "sans-serif"],
      },
    },
  },
  plugins: [],
};
