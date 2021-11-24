import { useParams } from "react-router";
import content from './content';
import './index.css';

function Post() {

  const { id } = useParams();
  const postcontent = content.filter(content => content.id === id);
  
  return (
    <div className="site">
      {postcontent.map((item) => {
        return (
          <div className="box">
            <p>{item.title}</p>
            <h1 className="lighter">{item.published}</h1>
            <h1>&nbsp;{item.tag}</h1>
            <h2>{item.artical}</h2>
          </div>
        ) 
      })}
    </div>
  );
}

export default Post;