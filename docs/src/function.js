import { useState, useEffect } from "react";
import { useParams } from "react-router";
import { db } from "./firebase";
import { collection, getDocs } from "firebase/firestore";
import { LineChart, Line, XAxis, YAxis, CartesianGrid } from "recharts";
import MathJax from "react-mathjax";
import "./index.css";

function Function() {
  const { version, func } = useParams();

  if (window.versions.includes(version)) {
    window.currentversion = version;
  }

  const [content, setContent] = useState([]);
  useEffect(() => {
    const getContent = async () => {
      const data = await getDocs(collection(db, "documentation"));
      setContent(data.docs.map((doc) => ({ ...doc.data(), id: doc.id })));
    };
    getContent();
  }, []);

  const contentfilteredbyfunc = content.filter(
    (content) => content.function === func
  );
  const contentfilteredbyversion = contentfilteredbyfunc.filter(
    (content) => content.version === window.currentversion
  );

  if (contentfilteredbyversion[0] !== undefined) {
    if (contentfilteredbyversion[0]["type"] === "computation") {
      return (
        <div>
          {contentfilteredbyversion.map((item) => {
            return (
              <div>
                <p className="p box">{item.function}</p>
                <p className="h1 bold">
                  {item.model}
                  <a href={item.sourcecode} className="h1 lighter link">
                    &nbsp;[source]
                  </a>
                </p>
                <p className="h1 lighter box">{item.description}</p> <br />
              </div>
            );
          })}
        </div>
      );
    }

    if (contentfilteredbyversion[0]["type"] === "activation") {
      const data = [];

      for (
        let i = 0;
        i < contentfilteredbyversion[0]["graphx"]["length"];
        i++
      ) {
        data.push({
          x: contentfilteredbyversion[0]["graphx"][i],
          y: contentfilteredbyversion[0]["graphy"][i],
        });
      }

      return (
        <div>
          {contentfilteredbyversion.map((item) => {
            return (
              <div>
                <p className="p box">{item.function}</p>
                <p className="h1 bold">
                  {item.model}
                  <a href={item.sourcecode} className="h1 lighter link">
                    &nbsp;[source]
                  </a>
                </p>
                <br />
                <p className="h1 lighter box">{item.description}</p> <br />
                <MathJax.Provider>
                  <div className="equation">
                    <p className="h1">{item.function}:</p>
                    <MathJax.Node formula={item.equation} className="latex" />
                    <p className="h1">{item.function} detivative:</p>
                    <MathJax.Node
                      formula={item.equationderivative}
                      className="latex"
                    />
                  </div>
                </MathJax.Provider>
                <div className="graph">
                  <LineChart
                    className="center"
                    width={300}
                    height={200}
                    data={data}
                    margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="x" />
                    <YAxis dataKey="y" />
                    <Line
                      type="monotone"
                      isAnimationActive={false}
                      dot={false}
                      dataKey="y"
                      stroke="#838383"
                    />
                  </LineChart>
                </div>
                <p className="h1">
                  parameters:
                  <br />
                  {item.parameters}
                </p>{" "}
                <br />
                <p className="h1">
                  examples:
                  <br />
                  {item.examples}
                </p>
                <pre>
                  <code>
                    import aiinpy as ai
                    x = [-1, -0.2, 0.5, 2, 100]
                  </code>
                </pre>
              </div>
            );
          })}
        </div>
      );
    }
  } else {
    return null;
  }
}

export default Function;
