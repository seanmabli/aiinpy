import { render } from 'react-dom';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import About from './about';
import Function from './function';
import './index.css';

window.currentversion = '0.0.17';
window.versions = ["0.0.17", "0.0.16", "0.0.13", "0.0.11"];

render(
  <BrowserRouter>
    <Routes>
      <Route exact path="/" element={<Navigate to={'/' + window.currentversion} replace />} />
      <Route exact path="/:version" element={<About />} />
      <Route exact path="/:version/:func" element={<Function />} />
    </Routes>
  </BrowserRouter>,
  document.getElementById('root'));