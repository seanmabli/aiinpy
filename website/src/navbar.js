import React, {useState} from 'react';
import { Link } from 'react-router-dom';
import content from './content';
import info from './info';
import './index.css';

function Navbar() {
  const [version, setVersion] = useState(info[0].currentversion);
  
  function getversion(val) {
    setVersion(val.target.value);
    if (info[0].possibleversions.includes(val.target.value) === true) {
      info[0].currentversion = val.target.value;
    }
  }

  return (
    <div>
      <div className="box">
        <Link to="/" className="p notext-decoration">aiinpy</Link> <br/>
      </div>
      <input type="text" value={version} onChange={getversion} className="h1 lighter version"/>
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