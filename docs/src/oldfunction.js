import { useState, useEffect } from 'react';
import { useParams } from "react-router";
import { db } from './firebase';
import { collection, getDocs } from 'firebase/firestore';
import Navbar from './navbar';
import './index.css';

function Function() {
  const { version, func } = useParams();

  console.log('function.js');
  if (window.versions.includes(version)) {
    window.currentversion = version;
  }

  const [content, setContent] = useState([]);
  const contentRef = collection(db, 'documentation')
  useEffect(() => {
    const getContent = async () => {
      const data = await getDocs(contentRef);
      setContent(data.docs.map((doc) => ({ ...doc.data(), id: doc.id})))
    };
    getContent();
  }, [])

  const contentfilteredbyfunc = content.filter(content => content.function === func);
  const contentfilteredbyversion = contentfilteredbyfunc.filter(content => content.version === window.currentversion);

  return (
    <div className="site">
      <div className="about"> 
        <Navbar />
      </div>
      <div className="function">
        {contentfilteredbyversion.map((item) => {
          return (
            <div>
              <p className="p box">{item.function}</p>
              <p className="h1 bold">{item.model}<a href={item.sourcecode} className="h1 lighter link">&nbsp;[source]</a></p>
              <p className="h1 lighter box">{item.discription}</p> <br />
            </div>
          ) 
        })}
      </div>
    </div>
  );
}

export default Function;
