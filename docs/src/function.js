import { useState, useEffect } from 'react';
import { useParams } from "react-router";
import { db } from './firebase';
import { collection, getDocs } from 'firebase/firestore';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './index.css';

function Function() {
  const { version, func } = useParams();

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
  const data = [
    {
      y: 0,
    },
    {
      y: 0.25,
    },
    {
      y: 0.5,
    },
    {
      y: 0.75,
    },
    {
      y: 1,
    },
    {
      y: 0.75,
    },
    {
      y: 0.5,
    },
    {
      y: 0.25,
    },
    {
      y: 0,
    },
  ];

  if (contentfilteredbyversion[0] !== undefined) {
    if (contentfilteredbyversion[0]['type'] === 'computation') {
      return (
        <div>
          {contentfilteredbyversion.map((item) => {
            return (
              <div>
                <p className="p box">{item.function}</p>
                <p className="h1 bold">{item.model}<a href={item.sourcecode} className="h1 lighter link">&nbsp;[source]</a></p>
                <p className="h1 lighter box">{item.description}</p> <br />
              </div>
            ) 
          })}
        </div>
      )
    }

    if (contentfilteredbyversion[0]['type'] === 'activation') {
      return (
        <div>
          {contentfilteredbyversion.map((item) => {
            return (
              <div>
                <p className="p box">{item.function}</p>
                <p className="h1 bold">{item.model}<a href={item.sourcecode} className="h1 lighter link">&nbsp;[source]</a></p>
                <p className="h1 lighter box">{item.description}</p> <br />

                <LineChart width={300} height={200} data={data} margin={{top: 5, right: 5, left: 5, bottom: 5}}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis allowDecimals={true} tickCount={5} domain={[-1, 1]} />
                  <YAxis dataKey="y" />
                  <Line type="monotone" dataKey="y" stroke="#838383" />
                </LineChart>
              </div>
            ) 
          })}
        </div>
      )
    }
  } else {
    return null
  }
}

export default Function;