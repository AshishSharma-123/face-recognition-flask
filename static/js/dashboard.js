document.addEventListener("DOMContentLoaded", function () {
    const createDatasetBtn = document.getElementById("createDatasetBtn");

    createDatasetBtn.addEventListener("click", () => {
        const datasetName = prompt("Enter the name of the dataset:");
        if (datasetName && datasetName.trim() !== "") {
            fetch("/create-dataset", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ name: datasetName })
            })
            .then(response => response.json())
            .then(data => {
                alert(data.message);
            })
            .catch(error => {
                console.error("Error:", error);
                alert("An error occurred while creating the dataset.");
            });
        } else {
            alert("Dataset name cannot be empty.");
        }
    });
});
// Show server-side flash message if present
document.addEventListener("DOMContentLoaded", function () {
    const messageBox = document.getElementById("messageBox");
    const serverMessage = messageBox?.dataset?.message;

    if (serverMessage) {
        alert(serverMessage);
    }
});

