import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import './App.css';

function Home() {
    const [topRatedBooks, setTopRatedBooks] = useState([]);

    useEffect(() => {
        const fetchTopRatedBooks = async () => {
            try {
                const response = await axios.get('http://127.0.0.1:5000/rating_based?top_n=10');
                setTopRatedBooks(response.data);
            } catch (error) {
                console.error("Error fetching top rated books:", error);
            }
        };

        fetchTopRatedBooks();
    }, []);

    return (
        <div>
            <h1>Book Store</h1>
            <h2>Top 10 Rated Books</h2>
            <br/><br/>
            <ul>
                {topRatedBooks.map((book, index) => (
                    <li key={index} className="book">
                        <img src={book.image_url} alt={book.book_title} style={{ width: '100px' }} />
                        <div className="book-details">
                            <div className="book-title">{book.book_title}</div>
                            <div className="book-author">by {book.book_author}</div>
                            <div className="book-rating">Avg Rating: {book.avg_rating}</div>
                            <div className="book-rating-count">User Count: {book.user_count}</div>
                            {/* Buy and Rate buttons */}
                            <div className="book-actions">
                                <button className="buy-button">Buy</button>
                                <div className="rate-section">
                                    <label htmlFor="rate">Rate: </label>
                                    <input
                                        type="number"
                                        id="rate"
                                        name="rate"
                                        min="0"
                                        max="10"
                                        defaultValue="0"
                                    />
                                    
                                </div>
                            </div>
                        </div>
                    </li>
                ))}
            </ul>
        </div>
    );
}


function Recommendations() {
    const [userId, setUserId] = useState('');
    const [topN, setTopN] = useState(5);
    const [bookName, setBookName] = useState('');
    const [recommendationType, setRecommendationType] = useState('content_based');
    const [books, setBooks] = useState([]);
    const [loading, setLoading] = useState(false);

    const fetchBooks = async () => {
        setLoading(true); // Set loading to true when fetching starts
        try {
            const response = await axios.get(`http://127.0.0.1:5000/${recommendationType}?user_id=${userId}&top_n=${topN}&book_name=${bookName}`);
            setBooks(response.data);
        } catch (error) {
            console.error("Error fetching data:", error);
        } finally {
            setLoading(false); // Set loading to false when fetching ends
        }
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        fetchBooks();
    };

    return (
        <div>
            <form onSubmit={handleSubmit}>
                <div>
                    <label>Recommendation Type:</label>
                    <select value={recommendationType} onChange={(e) => setRecommendationType(e.target.value)}>
                        <option value="content_based">Content-Based</option>
                        <option value="recommend">Collaborative</option>
                        <option value="hybrid">Hybrid</option>
                    </select>
                </div>
                {recommendationType === 'content_based' && (
                    <>
                        <div>
                            <label>Book Name:</label>
                            <input type="text" value={bookName} onChange={(e) => setBookName(e.target.value)} />
                        </div>
                        <div>
                            <label>Number of Recommendations:</label>
                            <input type="number" value={topN} onChange={(e) => setTopN(e.target.value)} />
                        </div>
                    </>
                )}
                {recommendationType === 'recommend' && (
                    <>
                        <div>
                            <label>User ID:</label>
                            <input type="text" value={userId} onChange={(e) => setUserId(e.target.value)} />
                        </div>
                        <div>
                            <label>Number of Recommendations:</label>
                            <input type="number" value={topN} onChange={(e) => setTopN(e.target.value)} />
                        </div>
                    </>
                )}
                {recommendationType === 'hybrid' && (
                    <>
                        <div>
                            <label>User ID:</label>
                            <input type="text" value={userId} onChange={(e) => setUserId(e.target.value)} />
                        </div>
                        <div>
                            <label>Book Name:</label>
                            <input type="text" value={bookName} onChange={(e) => setBookName(e.target.value)} />
                        </div>
                        <div>
                            <label>Number of Recommendations:</label>
                            <input type="number" value={topN} onChange={(e) => setTopN(e.target.value)} />
                        </div>
                    </>
                )}
                <button type="submit">Get Recommendations</button>
            </form>

            {loading ? (
                <div className="loading">Loading...</div> // Display loading indicator
            ) : (
                <>
                    <h2>Recommendations:</h2>
                    <br /><br />
                    <ul>
                        {books.map((book, index) => (
                            <li key={index} className="book">
                                <img src={book['Image-URL-M']} alt={book['Book-Title']} style={{ width: '100px' }} />
                                <div className="book-details">
                                    <div className="book-title">{book['Book-Title']}</div>
                                    <div className="book-author">by {book['Book-Author']}</div>
                                    
                                    {/* Buy and Rate buttons */}
                                    <div className="book-actions">
                                        <button className="buy-button">Buy</button>
                                        <div className="rate-section">
                                            <label htmlFor="rate">Rate: </label>
                                            <input
                                                type="number"
                                                id="rate"
                                                name="rate"
                                                min="0"
                                                max="10"
                                                defaultValue="0"
                                            />
                                            
                                        </div>
                                    </div>
                                </div>
                            </li>
                        ))}
                    </ul>
                </>
            )}
        </div>
    );
}



function App() {
    return (
        <Router>
            <div className="main-content">
                <nav className="sidebar">
                    <ul>
                        <li><Link to="/">Home</Link></li>
                        <li><Link to="/recommendations">Recommendations</Link></li>
                    </ul>
                </nav>
                <div className="App">
                    <Routes>
                        <Route exact path="/" element={<Home />} />
                        <Route path="/recommendations" element={<Recommendations />} />
                    </Routes>
                </div>
            </div>
        </Router>
    );
}

export default App;

