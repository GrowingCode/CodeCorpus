package com.mingrisoft.DOM_demo;

import java.io.File;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;

import com.mingrisoft.entity.Book;
import com.mingrisoft.entity.BookPrice;

/**
 * @author bwm
 * 
 */
public class BookDOMBulid {

	/**
	 * 预置数据
	 * 
	 * @return
	 */
	protected List<Book> initData() {
		List<Book> bookList = new ArrayList<Book>();
		Book book;
		BookPrice bookPrice;
		Map<String, String> attributeMap;

		// 第一本书
		book = new Book();
		bookPrice = new BookPrice();
		book.setName("《C#从入门到精通(第2版)》");
		book.setPublisher("清华大学出版社");
		book.setCompany("明日科技");
		book.setAuthor("王小科");
		book.setIsbn("9787302226628");

		attributeMap = new HashMap<String, String>();
		attributeMap.put("unit", "yuan");
		attributeMap.put("unitType", "RMB");
		bookPrice.setAttributeMap(attributeMap);
		bookPrice.setValue(Double.parseDouble("69.80"));
		book.setPrice(bookPrice);
		try {
			book.setUrl(new URL("http://www.mingribook.com/bookinfo.php?id=227"));
		} catch (MalformedURLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		bookList.add(book);

		// 第二本书
		book = new Book();
		bookPrice = new BookPrice();
		book.setName("《JavaScript开发技术大全》");
		book.setPublisher("人民邮电出版社");
		book.setCompany("明日科技");
		book.setAuthor("梁冰");
		book.setIsbn("9787115179708");

		attributeMap = new HashMap<String, String>();
		attributeMap.put("unit", "yuan");
		attributeMap.put("unitType", "RMB");
		bookPrice.setAttributeMap(attributeMap);
		bookPrice.setValue(Double.parseDouble("65.00"));
		book.setPrice(bookPrice);
		try {
			book.setUrl(new URL("http://www.mingribook.com/bookinfo.php?id=138"));
		} catch (MalformedURLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		bookList.add(book);

		// 第三本书
		// ......

		return bookList;

	}

	/**
	 * 创建XML文件
	 * 
	 * @param node
	 *            DOM节点
	 * @param url
	 *            XML文件路径
	 */
	public void writeFile(Node node, String url) {
		TransformerFactory transformerFactory = TransformerFactory
				.newInstance();
		DOMSource domSource = new DOMSource(node);
		StreamResult streamResult = new StreamResult(new File(url));
		try {
			Transformer transformer = transformerFactory.newTransformer();
			transformer.transform(domSource, streamResult);
		} catch (TransformerConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (TransformerException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	/**
	 * 创建XML
	 * 
	 * @return
	 */
	public Document bulid() {
		DocumentBuilderFactory documentBuilderFactory = DocumentBuilderFactory
				.newInstance();
		DocumentBuilder dombuilder = null;
		Document document = null;
		try {
			dombuilder = documentBuilderFactory.newDocumentBuilder();
		} catch (ParserConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		List<Book> bookList =initData();
		System.out.println(bookList);
		if (bookList != null) {

			document = dombuilder.newDocument();
			Element books = document.createElement("book:books");
			books.setAttribute("xmlns:book", "http://www.mingrisoft.com");
			for (Iterator iterator = bookList.iterator(); iterator.hasNext();) {
				Book bookEntity = (Book) iterator.next();

				Element book = document.createElement("book:book");
				Element name = document.createElement("book:name");
				Element publisher = document.createElement("book:publisher");
				Element company = document.createElement("book:company");
				Element author = document.createElement("book:author");
				Element isbn = document.createElement("book:ISBN");
				Element price = document.createElement("book:price");
				Element url = document.createElement("book:url");

				name.setTextContent(bookEntity.getName());
				publisher.setTextContent(bookEntity.getPublisher());
				company.setTextContent(bookEntity.getCompany());
				author.setTextContent(bookEntity.getAuthor());
				isbn.setTextContent(bookEntity.getIsbn());
				price.setTextContent(bookEntity.getPrice().getValue()
						.toString());
				url.setTextContent(bookEntity.getUrl().toString());

				book.appendChild(name);
				book.appendChild(publisher);
				book.appendChild(company);
				book.appendChild(author);
				book.appendChild(isbn);
				book.appendChild(price);
				book.appendChild(url);

				books.appendChild(book);
			}
			document.appendChild(books);
		}

		return document;
	}

	public static void main(String[] arg) {

		String pathname = "xmldemo/BookDOMBulid.xml";
		BookDOMBulid bulid = new BookDOMBulid();
		bulid.writeFile(bulid.bulid(), pathname);

	}

}
