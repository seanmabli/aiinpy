import ReactDOM from 'react-dom';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import About from './about';
import Function from './function';
import './index.css';

window.currentversion = '0.0.17';
window.versions = ["0.0.17", "0.0.16", "0.0.13", "0.0.11"];

ReactDOM.render(
  <Router>
    <Routes>
      <Route exact path="/" element={About} />
      <Route exact path="/:version" element={About} />
      <Route exact path="/:version/:func" element={Function} />
    </Routes>
  </Router>,

document.getElementById('root'));