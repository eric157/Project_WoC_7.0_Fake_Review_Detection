async function analyzeReviews() {
    const productUrl = document.getElementById('productUrl').value;
    const resultsDiv = document.getElementById('reviews-list');
    const summaryDiv = document.getElementById('summary');
    const avgRatingSpan = document.getElementById('averageRating');
    const reviewCountSpan = document.getElementById('reviewCount');

    if (!productUrl) {
        alert("Please enter a product URL.");
        return;
    }

    resultsDiv.innerHTML = `<p class="loading-message-container">Analyzing reviews... <i class='fas fa-circle-notch fa-spin loading-icon'></i></p>`; // Wrapped in container
    summaryDiv.style.display = 'none';

    try {
        const response = await fetch('http://127.0.0.1:5000/scrape_and_predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ product_url: productUrl })
        });

        if (!response.ok) {
            const message = `Error: ${response.status}`;
            throw new Error(message);
        }

        const data = await response.json();
        if (data.reviews_data && data.reviews_data.length > 0) {
            displayResults(data.reviews_data, resultsDiv);
            displaySummary(data.reviews_data, avgRatingSpan, reviewCountSpan, summaryDiv);
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
            const predictionText = reviewData.prediction === '1.0' ? 'Fake' : (reviewData.prediction === '0.0' ? 'Real' : reviewData.prediction);
            html += `
                <div class="review-item">
                    <p class="review-text">"${reviewData.review}"</p>
                    <p class="review-rating">Rating: ${reviewData.rating}</p>
                    <p class="prediction ${predictionText.toLowerCase().replace(' ', '-')}">Prediction: <span>${predictionText}</span></p>
                </div>
            `;
        });
    }
    resultsDiv.innerHTML = html;
}


function displaySummary(reviews, avgRatingSpan, reviewCountSpan, summaryDiv) {
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
}