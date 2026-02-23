document.addEventListener("DOMContentLoaded", () => {
  const items = document.querySelectorAll(".stPlotlyChart");
  items.forEach((item, idx) => {
    item.animate(
      [
        { opacity: 0, transform: "translateY(8px)" },
        { opacity: 1, transform: "translateY(0)" }
      ],
      { duration: 400 + idx * 80, easing: "ease-out" }
    );
  });
});
