import { Link } from 'react-router-dom';
import Dropdown from 'react-bootstrap/Dropdown';
import content from './content';
import './index.css';

function about() {
  return(
    <div className="site">
      <div className="about">
        <div className="box">
          <Link to="/" className="p notext-decoration">aiinpy</Link>
          <Dropdown>
            <Dropdown.Toggle variant="success">
              0.0.16
            </Dropdown.Toggle>
            
            <Dropdown.Menu>
              <Dropdown.Item href="#/action-1">Action</Dropdown.Item>
              <Dropdown.Item href="#/action-2">Another action</Dropdown.Item>
              <Dropdown.Item href="#/action-3">Something else</Dropdown.Item>
            </Dropdown.Menu>
          </Dropdown>
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