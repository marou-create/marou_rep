const inputs = document.querySelectorAll(".input");

inputs.forEach((input) => {
  input.onchange = function () {
    if (input.value !== "") {
      input.classList.add("filled");
    } else {
      input.classList.remove("filled");
    }
  };
});
