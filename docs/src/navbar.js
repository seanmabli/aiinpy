import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { db } from './firebase'
import './index.css';
import { collection, getDocs } from 'firebase/firestore';

function Navbar() {
  const [version, setVersion] = useState(window.currentversion);

  function getversion(val) {
    setVersion(val.target.value);
    window.currentversion = val.target.value;
  }

  const [content, setContent] = useState([]);
  useEffect(() => {
    const getContent = async () => {
      const data = await getDocs(collection(db, 'documentation'));
      setContent(data.docs.map((doc) => ({ ...doc.data(), id: doc.id})))
    };
    getContent();
  }, [])

  const contentfilteredbyversion = content.filter(content => content.version === window.currentversion);

  return (
    <div>
      <div className="box">
        <Link to='/' className="p notext-decoration">aiinpy</Link> <br/>
      </div>

      <div class="inlinebox">
        <label className="h1" >version:&nbsp; 
          <select className="h1 lighter version" value={version} onChange={getversion}>
          {window.versions.map((option) => (<option>{option}</option>))}
          </select>
        </label>
      </div>

      {contentfilteredbyversion.map((item) => {
        return (
          <div>
            <Link to={item.url} className="h1 lighter link"> {item.title} </Link> <br />
          </div>
        )
      })}
      <div className="box">
        <a href="https://github.com/seanmabli/aiinpy" className="h1 lighter link">github</a> 
        <a href="https://pypi.org/project/aiinpy/" className="h1 lighter link">&nbsp;pypi</a> 
      </div>
    </div>
  );
}

export default Navbar;
