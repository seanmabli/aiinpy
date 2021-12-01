import { useParams } from "react-router";
import Navbar from './navbar';
import content from './content';
import info from './version';
import './index.css';

function Function() {

  const { id } = useParams();
  const contentfiltedbyid = content.filter(content => content.id === id);
  const contentfiltedbyversion = contentfiltedbyid.filter(content => content.version === info[0].currentversion);
  
  return (
    <div className="site">
      <div className="about"> 
        <Navbar />
      </div>
      <div className="function">
        {contentfiltedbyversion.map((item) => {
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