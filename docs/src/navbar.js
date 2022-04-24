import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { db } from "./firebase";
import "./index.css";
import { collection, getDocs } from "firebase/firestore";

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);

  const toggling = () => setIsOpen(!isOpen);

  const onOptionClicked = (value) => () => {
    setIsOpen(false);
    window.currentversion = value;
  };

  const [content, setContent] = useState([]);
  useEffect(() => {
    const getContent = async () => {
      const data = await getDocs(collection(db, "documentation"));
      setContent(data.docs.map((doc) => ({ ...doc.data(), id: doc.id })));
    };
    getContent();
  }, []);

  const contentfilteredbyversion = content.filter(
    (content) => content.version === window.currentversion
  );

  return (
    <div>
      <div className="box">
        <Link to={"/"+window.currentversion} className="p notext-decoration">
          aiinpy
        </Link>
        <br />
      </div>

      <div className="dropdowncontainer" onClick={toggling}>
        <div className="h1">version:&nbsp;{window.currentversion}</div>
        {isOpen && (
          <div className="dropdownlistcontainer">
            <ul className="dropdownlist">
              {window.versions.map((option) => (
                <li className="dropdown" onClick={onOptionClicked(option)}>
                  {option}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {contentfilteredbyversion.map((item) => {
        return (
          <div>
            <Link to={item.url} className="h1 lighter link">
              {item.title}
            </Link>
            <br />
          </div>
        );
      })}
      <div className="box">
        <a
          href="https://github.com/seanmabli/aiinpy"
          className="h1 lighter link"
        >
          github
        </a>
        <a href="https://pypi.org/project/aiinpy/" className="h1 lighter link">
          &nbsp;pypi
        </a>
      </div>
    </div>
  );
}
