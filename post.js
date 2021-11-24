import { useParams } from "react-router";
import { Link } from 'react-router-dom';
import content from './content';
import './index.css';

function Post() {

  const { id } = useParams();
  const postcontent = content.filter(content => content.id === id);

  return (
    <div className="site">
      <div className="about">
        <Link to="/" className="p notext-decoration">aiinpy</Link>
        {content.map((item) => {
          return (
            <div>
              <Link to={item.url} className="h2 link"> {item.title} </Link> <br />
            </div>
          )
        })}
      </div>
      <div className="blog">
        {postcontent.map((item) => {
          return (
            <p>{item.title}</p>
          ) 
        })}
      </div>
    </div>
  );
}

export default Post;