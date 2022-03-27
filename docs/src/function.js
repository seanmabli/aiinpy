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
      x: 4000,
      y: 2400,
    },
    {
      x: 3000,
      y: 1398,
    },
    {
      x: 2000,
      y: 9800,
    },
    {
      x: 2780,
      y: 3908,
    },
    {
      x: 1890,
      y: 4800,
    },
    {
      x: 2390,
      y: 3800,
    },
    {
      x: 3490,
      y: 4300,
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

                <LineChart width={500} height={300} data={data} margin={{top: 5, right: 30, left: 20, bottom: 5}}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="x" />
                  <YAxis dataKey="y" />
                  <Line type="monotone" dataKey="x" stroke="#8884d8" activeDot={{ r: 8 }} />
                  <Line type="monotone" dataKey="y" stroke="#82ca9d" />
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