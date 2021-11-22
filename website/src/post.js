import { useParams } from "react-router";
import { Link } from 'react-router-dom';
import pages from './content.json';
import './index.css';


function Post() {

  const { id } = useParams();
  const postcontent = pages.filter(pages => pages.url === id);

  return (
    <div className="site">
      <div className="about">
        <Link to="/" className="p notext-decoration">aiinpy</Link>
        {pages.map((item) => {
          return (
            <div>
              <Link to={item.url} className="h2 link"> {item.title} </Link> <br />
            </div>
          ) 
        })}
      </div>
    </div>
  );
}
/*
      {postcontent.map((item) => {
        return (
        ) 
      })}
      */
export default Post;