import { Link } from 'react-router-dom';
import content from './content';
import './index.css';

function about() {
  return(
    <div className="site">
      <div className="about">
        <div className="box">
          <Link to="/" className="p notext-decoration">aiinpy</Link>
          <p className="h1">0.0.16</p>
        </div>
        {content.map((item) => {
          return (
            <div>
              <Link to={item.url} className="h1 lighter link"> {item.title} </Link> <br />
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default about;