import { render } from "react-dom";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import About from "./about";
import Function from "./function";
import Navbar from "./navbar";
import "./index.css";

window.currentversion = "0.0.18";
window.versions = ["0.0.18", "0.0.17", "0.0.16", "0.0.13", "0.0.11"];

render(
  <BrowserRouter>
    <div className="site">
      <div className="about">
        <Navbar />
      </div>
      <div className="function">
        <Routes>
          <Route
            exact
            path="/"
            element={<Navigate to={"/" + window.currentversion} replace />}
          />
          <Route exact path="/:version" element={<About />} />
          <Route exact path="/:version/:func" element={<Function />} />
        </Routes>
      </div>
    </div>
  </BrowserRouter>,
  document.getElementById("root")
);
