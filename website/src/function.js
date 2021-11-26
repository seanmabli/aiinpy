import { useParams } from "react-router";
import { Link } from "react-router-dom";
import content from './content';
import './index.css';

function Function() {

  const { id } = useParams();
  const postcontent = content.filter(content => content.id === id);
  
  return (
    <div className="site">
      <div className="about">
        <div className="box">
          <Link to="/" className="p notext-decoration">aiinpy</Link> <br />
        </div>
        <p className="h1 lighter">0.0.16</p>
        {content.map((item) => {
          return (
            <div>
              <Link to={item.url} className="h1 lighter notext-decoration"> {item.title} </Link> <br />
            </div>
          )
        })}
      </div>
      <div className="function">
        {postcontent.map((item) => {
          return (
            <div>
              <div className="box">
                <p className="p">{item.id}</p>
              </div>
              <p className="h1 lighter">{item.model}</p> <br />
            </div>
          ) 
        })}
      </div>
    </div>
  );
}

export default Function;