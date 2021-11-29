import React, {useState} from 'react';
import { Link } from 'react-router-dom';
import content from './content';
import info from './info';
import './index.css';

function Navbar() {
  const version = info[0].version;
  
  function setversion(val) {
    info[0].version = val.target.value;
  }

  return (
    <div>
      <div className="box">
        <Link to="/" className="p notext-decoration">aiinpy</Link> <br/>
        <h1>{version}</h1>
        <input type="text" value={info[0].version} onChange={setversion} />
      </div>
      {content.map((item) => {
        return (
          <div>
            <Link to={item.url} className="h1 lighter link"> {item.title} </Link> <br />
          </div>
        )
      })}
    </div>
  );
}

export default Navbar;