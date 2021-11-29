import { useParams } from "react-router";
import Navbar from './navbar';
import content from './content';
import './index.css';

function Function() {

  const { id } = useParams();
  const postcontent = content.filter(content => content.id === id);
  
  return (
    <div className="site">
      <div className="about"> 
        <Navbar />
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