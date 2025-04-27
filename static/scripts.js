document.addEventListener("DOMContentLoaded", function () {
    const inputs = document.querySelectorAll(".image-block input");

    inputs.forEach((input) => {
        input.addEventListener("focus", function () {
            input.nextElementSibling.classList.add("active");
        });

        input.addEventListener("blur", function () {
            if (input.value === "") {
                input.nextElementSibling.classList.remove("active");
            }
        });
    });
});