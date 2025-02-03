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
    if (!numReviews || isNaN(numReviews) || numReviews <= 0 ) {
        alert("Please enter a valid number of reviews.");
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

function displayResults(reviews, resultsDiv) {
    let html = "";
    if (!reviews || reviews.length === 0) {
        html = "<p>No reviews available.</p>";
    } else {
        html += "<h2>Reviews:</h2>";
        reviews.forEach(reviewData => {
            const predictionText = reviewData.prediction;
            const probability = reviewData.probability === 'null' ? "N/A" : parseFloat(reviewData.probability).toFixed(2);
            html += `
                <div class="review-item">
                    <p class="review-text">"${reviewData.review}"</p>
                    <p class="review-rating">Rating: ${reviewData.rating}</p>
                    <p class="prediction ${predictionText.toLowerCase().replace(' ', '-')}">Prediction: <span>${predictionText}</span></p>
                     <p class="probability">Probability: <span>${probability}</span></p>
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
    inferenceTimeSpan.textContent =  inferenceTime.toFixed(3) + 's'; // Added inference Time display

}