const header = document.querySelector("header");
const headerMenu = document.querySelector(".header_menu"); // Add a dot before 'header_menu'
const menuBtn = document.querySelector(".menu-btn");
const headerMenuItems = headerMenu.querySelectorAll("li a");

window.addEventListener("scroll", () => {
    header.classList.toggle("sticky", window.scrollY > 0);
});

header.classList.add("sticky");

menuBtn.addEventListener("click", () => {
    headerMenu.classList.toggle("show");
});

headerMenuItems.forEach((item) => { // Add an opening parenthesis here
    item.addEventListener("click", () => {
        headerMenu.classList.remove("show");
    });
});

// slider//

