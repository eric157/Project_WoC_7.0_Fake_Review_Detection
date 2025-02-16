async function analyzeReviews() {
    const productUrl = document.getElementById('productUrl').value;
    const numReviews = document.getElementById('numReviews').value;
    const resultsDiv = document.getElementById('reviews-list');
    const summaryDiv = document.getElementById('summary');
    const avgRatingSpan = document.getElementById('averageRating');
    const reviewCountSpan = document.getElementById('reviewCount');
    const fakePercentageSpan = document.getElementById('fakePercentage');
    const inferenceTimeSpan= document.getElementById('inferenceTime'); // Added inferenceTime

    if (!productUrl) {
        alert("Please enter a product URL.");
        return;
    }
    if (!numReviews || isNaN(numReviews) || numReviews <= 0 || numReviews > 100) { // Added max reviews validation
        alert("Please enter a valid number of reviews between 1 and 100."); // Updated alert message
        return;
    }

    resultsDiv.innerHTML = `<p class="loading-message-container">Analyzing reviews... <i class='fas fa-circle-notch fa-spin loading-icon'></i></p>`;
    summaryDiv.style.display = 'none';

    try {
        const response = await fetch('http://127.0.0.1:5000/scrape_and_predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ product_url: productUrl, max_reviews: parseInt(numReviews, 10) })
        });

        if (!response.ok) {
            const message = `Error: ${response.status}`;
            throw new Error(message);
        }

        const data = await response.json();
        if (data.reviews_data && data.reviews_data.length > 0) {
            displayResults(data.reviews_data, resultsDiv);
            displaySummary(data.reviews_data, avgRatingSpan, reviewCountSpan, summaryDiv, data.fake_percentage, fakePercentageSpan,data.inference_time, inferenceTimeSpan); // Added inferenceTime here also
            summaryDiv.style.display = 'block';
        } else {
            resultsDiv.innerHTML = "<p>No reviews found or an error occurred during scraping.</p>";
            summaryDiv.style.display = 'none';
        }
    } catch (error) {
        resultsDiv.innerHTML = `<p class="error">Error analyzing reviews: ${error.message}</p>`;
        summaryDiv.style.display = 'none';
        console.error("Fetch error:", error);
    }
}

async function predictManualReview() {
    const reviewText = document.getElementById('reviewText').value;
    const rating = document.getElementById('manualRating').value;
    const resultDiv = document.getElementById('manual-prediction-result');
    const predictionTextSpan = document.getElementById('manual-prediction-text');
    const probabilitySpan = document.getElementById('manual-prediction-probability');

    if (!reviewText) {
        alert("Please enter review text.");
        return;
    }
    if (!rating || isNaN(rating) || rating < 1 || rating > 5) {
        alert("Please enter a rating between 1 and 5.");
        return;
    }

    resultDiv.style.display = 'none';
    predictionTextSpan.textContent = ``; // Removed "Predicting..." text
    probabilitySpan.textContent = '';
    resultDiv.style.opacity = '0'; // Start with opacity 0 for animation
    resultDiv.style.display = 'block'; // Show result div

    // Fade-in animation for result
    setTimeout(() => {
        resultDiv.style.transition = 'opacity 0.5s ease-in-out';
        resultDiv.style.opacity = '1';
    }, 10); // Small delay to ensure transition is applied

    try {
        const response = await fetch('http://127.0.0.1:5000/predict_manual_review', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ review_text: reviewText, rating: parseFloat(rating) })
        });

        if (!response.ok) {
            const message = `Error: ${response.status}`;
            throw new Error(message);
        }

        const data = await response.json();
        predictionTextSpan.textContent = `${data.prediction}`; // Just Prediction value
        probabilitySpan.textContent = `${data.probability === 'null' ? 'N/A' : parseFloat(data.probability).toFixed(2)}`; // Just Confidence value
        if (data.prediction === "Fake") {
            predictionTextSpan.className = 'prediction fake';
        } else if (data.prediction === "Real") {
            predictionTextSpan.className = 'prediction real';
        } else {
            predictionTextSpan.className = 'prediction language-error-prediction-failed'; // Or another appropriate class
        }


    } catch (error) {
        predictionTextSpan.textContent = `Error: ${error.message}`;
        probabilitySpan.textContent = '';
        predictionTextSpan.className = 'prediction prediction-error';
        console.error("Fetch error:", error);
    }
}


function displayResults(reviews, resultsDiv) {
    let html = "";
    if (!reviews || reviews.length === 0) {
        html = "<p>No reviews available.</p>";
    } else {
        html += "<h2>Scanned Reviews:</h2>"; // Updated heading for reviews list
        reviews.forEach(reviewData => {
            const predictionText = reviewData.prediction;
            const probability = reviewData.probability === 'null' ? "N/A" : parseFloat(reviewData.probability).toFixed(2);
            html += `
                <div class="review-item">
                    <p class="review-text">"${reviewData.review}"</p>
                    <div class="review-meta"> <!-- Review Meta Container -->
                        <p class="review-rating">Rating: <span>${reviewData.rating}</span></p>
                        <p class="prediction ${predictionText.toLowerCase().replace(' ', '-')}">Prediction: <span>${predictionText}</span></p>
                        <p class="probability">Confidence: <span>${probability}</span></p>
                    </div>
                </div>
            `;
        });
    }
    resultsDiv.innerHTML = html;
}


function displaySummary(reviews, avgRatingSpan, reviewCountSpan, summaryDiv, fakePercentage, fakePercentageSpan, inferenceTime, inferenceTimeSpan) {
    if (!reviews || reviews.length === 0) {
        summaryDiv.style.display = 'none';
        return;
    }

    let totalRating = 0;

    reviews.forEach(review => {
        if (review.rating && !isNaN(parseFloat(review.rating))) {
            totalRating += parseFloat(review.rating);
        }
    });

    const averageRating = reviews.length > 0 ? (totalRating / reviews.length).toFixed(2) : 'N/A';

    avgRatingSpan.textContent = averageRating;
    reviewCountSpan.textContent = reviews.length;
    fakePercentageSpan.textContent = fakePercentage.toFixed(2) + '%';
    inferenceTimeSpan.textContent =  inferenceTime.toFixed(3) + 's';

}

function openTab(tabId) {
    const tabContents = document.querySelectorAll('.tab-content');
    const tabButtons = document.querySelectorAll('.tab-button');

    tabContents.forEach(content => {
        content.classList.remove('active-content');
    });

    tabButtons.forEach(button => {
        button.classList.remove('active');
    });

    document.getElementById(tabId).classList.add('active-content');
    document.querySelector(`.tab-button[onclick="openTab('${tabId}')"]`).classList.add('active');
}