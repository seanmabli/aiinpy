import { Link } from 'react-router-dom';
import content from './content.json';
import './index.css';

function about() {
  return(
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
    </div>
  )
}

export default about;